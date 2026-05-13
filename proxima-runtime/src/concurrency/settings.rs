//! [`ConcurrencySettings`] — the conflaguration projection of [`Concurrency`],
//! first-class alongside the fluent [`ConcurrencyBuilder`](super::ConcurrencyBuilder)
//! (P4). Names builtins only; a foreign law/signal is reachable from config via an
//! on-demand registry (not pre-built) — out of this struct's scope by design.
//! Mirrors prime's `PrimeConfig` (`#[setting(default_str, resolve_with)]`) and
//! proxima-h1's `ResponseHandlingConfig` (`#[setting(skip)]` for serde-only
//! fields).

use core::fmt;
use core::str::FromStr;
use core::time::Duration;

use conflaguration::{Settings, Validate};
use serde::{Deserialize, Serialize};

use super::Bounds;
use super::strategy::{Concurrency, Preset};

/// Parse error for concurrency config fields (conflaguration's `resolve_with`
/// demands a `std::error::Error` parser error type).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseError(alloc::string::String);

impl fmt::Display for ParseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for ParseError {}

/// A control window as a human duration: `"150ms"`, `"2s"`, `"500us"`, `"1m"`.
/// `FromStr` drives the conflaguration/env path; serde (de)serialises the same
/// string form so TOML reads `window = "150ms"`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Window(pub Duration);

impl Window {
    #[must_use]
    pub fn duration(self) -> Duration {
        self.0
    }
}

impl FromStr for Window {
    type Err = ParseError;

    fn from_str(input: &str) -> Result<Self, Self::Err> {
        let trimmed = input.trim();
        let (digits, unit) = trimmed
            .find(|character: char| character.is_ascii_alphabetic())
            .map_or((trimmed, "ms"), |index| trimmed.split_at(index));
        let value: u64 = digits.trim().parse().map_err(|err| {
            ParseError(alloc::format!(
                "window: '{digits}' is not an integer: {err}"
            ))
        })?;
        let duration = match unit.trim() {
            "us" | "µs" => Duration::from_micros(value),
            "ms" | "" => Duration::from_millis(value),
            "s" => Duration::from_secs(value),
            "m" => Duration::from_secs(value * 60),
            other => {
                return Err(ParseError(alloc::format!(
                    "window: '{other}' is not us|ms|s|m"
                )));
            }
        };
        Ok(Self(duration))
    }
}

impl Serialize for Window {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&alloc::format!("{}ms", self.0.as_millis()))
    }
}

impl<'de> Deserialize<'de> for Window {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let text = alloc::string::String::deserialize(deserializer)?;
        text.parse().map_err(serde::de::Error::custom)
    }
}

/// The preset name (the value-carrying part comes from `fixed`/`target_ms`/
/// `ceiling`). Serde (de)serialises through [`FromStr`] / the canonical token so
/// the TOML, env, and fluent spellings are identical — no `hillclimb` vs
/// `hill_climb` drift between the conflaguration and serde paths.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PresetName {
    Fixed,
    HillClimb,
    Gradient,
    LatencyTarget,
    Headroom,
}

impl PresetName {
    #[must_use]
    pub fn token(self) -> &'static str {
        match self {
            Self::Fixed => "fixed",
            Self::HillClimb => "hillclimb",
            Self::Gradient => "gradient",
            Self::LatencyTarget => "latency_target",
            Self::Headroom => "headroom",
        }
    }
}

impl FromStr for PresetName {
    type Err = ParseError;

    fn from_str(input: &str) -> Result<Self, Self::Err> {
        match input.trim().to_ascii_lowercase().as_str() {
            "fixed" => Ok(Self::Fixed),
            "hillclimb" | "hill_climb" => Ok(Self::HillClimb),
            "gradient" | "" => Ok(Self::Gradient),
            "latency_target" | "latency" => Ok(Self::LatencyTarget),
            "headroom" => Ok(Self::Headroom),
            other => Err(ParseError(alloc::format!(
                "preset: '{other}' is not fixed|hillclimb|gradient|latency_target|headroom"
            ))),
        }
    }
}

impl Serialize for PresetName {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.token())
    }
}

impl<'de> Deserialize<'de> for PresetName {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let text = alloc::string::String::deserialize(deserializer)?;
        text.parse().map_err(serde::de::Error::custom)
    }
}

fn parse_preset(value: &str) -> Result<PresetName, ParseError> {
    PresetName::from_str(value)
}

fn parse_window(value: &str) -> Result<Window, ParseError> {
    Window::from_str(value)
}

/// `[min, max]` concurrency bounds. A newtype so the `#[setting(skip)]` path
/// (conflaguration fills skipped fields from `Default`, not the serde default)
/// still yields `[1, 512]` rather than `[0, 0]`. `#[serde(transparent)]` keeps the
/// TOML form a bare array: `bounds = [1, 512]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct BoundsPair(pub [usize; 2]);

impl Default for BoundsPair {
    fn default() -> Self {
        Self([1, 512])
    }
}

fn default_bounds() -> BoundsPair {
    BoundsPair::default()
}

/// Typed, file- and env-driven config for [`Concurrency`]. Resolve via
/// [`Concurrency::from_settings`].
///
/// ```toml
/// [concurrency]
/// preset = "gradient"
/// coefficient_of_variation_threshold = 0.05   # act on >5% relative moves
/// window = "150ms"
/// bounds = [1, 512]
/// ```
#[derive(Debug, Clone, PartialEq, Settings, Serialize, Deserialize)]
#[settings(prefix = "CONCURRENCY")]
pub struct ConcurrencySettings {
    /// Named strategy. Default `gradient` at the library level; surfaces with a
    /// different intent (rekt) construct this with `hillclimb` explicitly.
    #[setting(default_str = "gradient", resolve_with = "parse_preset")]
    #[serde(default = "default_preset")]
    pub preset: PresetName,

    /// Cap for `preset = "fixed"`.
    #[setting(default = 25)]
    #[serde(default = "default_fixed")]
    pub fixed: usize,

    /// p99 target in milliseconds for `preset = "latency_target"`.
    #[setting(default = 5.0)]
    #[serde(default = "default_target_ms")]
    pub target_ms: f64,

    /// Utilisation ceiling for `preset = "headroom"`.
    #[setting(default = 0.85)]
    #[serde(default = "default_ceiling")]
    pub ceiling: f64,

    /// Minimum relative signal change to act on, a fraction (`0.05` = 5%);
    /// smaller moves are treated as noise. Same units as the workload's measured
    /// coefficient of variation. See [`Gate`](super::Gate).
    #[setting(default = 0.05)]
    #[serde(default = "default_coefficient_of_variation_threshold")]
    pub coefficient_of_variation_threshold: f64,

    /// Control window cadence (`"150ms"`).
    #[setting(default_str = "150ms", resolve_with = "parse_window")]
    #[serde(default = "default_window")]
    pub window: Window,

    /// Re-measure this many windows before acting.
    #[setting(default = 1)]
    #[serde(default = "default_reprobe")]
    pub reprobe: u32,

    /// `[min, max]` concurrency. Serde/TOML only (an array has no env scalar
    /// form); env-driven deployments keep the default and tune via the file.
    #[setting(skip)]
    #[serde(default = "default_bounds")]
    pub bounds: BoundsPair,

    /// Seed concurrency.
    #[setting(default = 16)]
    #[serde(default = "default_start")]
    pub start: usize,
}

fn default_start() -> usize {
    16
}

fn default_preset() -> PresetName {
    PresetName::Gradient
}
fn default_fixed() -> usize {
    25
}
fn default_target_ms() -> f64 {
    5.0
}
fn default_ceiling() -> f64 {
    0.85
}
fn default_coefficient_of_variation_threshold() -> f64 {
    0.05
}
fn default_window() -> Window {
    Window(Duration::from_millis(150))
}
fn default_reprobe() -> u32 {
    1
}

impl Default for ConcurrencySettings {
    fn default() -> Self {
        Self {
            preset: PresetName::Gradient,
            fixed: 25,
            target_ms: 5.0,
            ceiling: 0.85,
            coefficient_of_variation_threshold: 0.05,
            window: default_window(),
            reprobe: 1,
            bounds: default_bounds(),
            start: default_start(),
        }
    }
}

impl Validate for ConcurrencySettings {
    fn validate(&self) -> conflaguration::Result<()> {
        let mut errors = alloc::vec::Vec::new();
        if self.bounds.0[0] < 1 || self.bounds.0[0] > self.bounds.0[1] {
            errors.push(conflaguration::ValidationMessage::new(
                "bounds",
                "require 1 <= min <= max",
            ));
        }
        if self.start < self.bounds.0[0] || self.start > self.bounds.0[1] {
            errors.push(conflaguration::ValidationMessage::new(
                "start",
                "must lie within [min, max]",
            ));
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(conflaguration::Error::Validation { errors })
        }
    }
}

impl ConcurrencySettings {
    /// The named-strategy form this settings resolves to.
    #[must_use]
    pub fn to_preset(&self) -> Preset {
        match self.preset {
            PresetName::Fixed => Preset::Fixed(self.fixed),
            PresetName::HillClimb => Preset::HillClimb,
            PresetName::Gradient => Preset::Gradient,
            PresetName::LatencyTarget => {
                Preset::LatencyTarget(Duration::from_micros((self.target_ms * 1_000.0) as u64))
            }
            PresetName::Headroom => Preset::Headroom(self.ceiling),
        }
    }

    /// The `Bounds` this settings carries.
    #[must_use]
    pub fn to_bounds(&self) -> Bounds {
        Bounds {
            min: self.bounds.0[0],
            max: self.bounds.0[1],
            start: self.start,
        }
    }
}

impl Concurrency {
    /// Resolve a [`ConcurrencySettings`] into the runtime knob. The config path
    /// of P4 — must agree byte-for-byte (via the descriptor) with the equivalent
    /// fluent build.
    pub fn from_settings(settings: &ConcurrencySettings) -> Result<Self, &'static str> {
        let mut builder = Self::builder()
            .preset(settings.to_preset())
            .coefficient_of_variation_threshold(settings.coefficient_of_variation_threshold)
            .window(settings.window.duration())
            .reprobe(settings.reprobe)
            .bounds_full(settings.to_bounds());
        // a foreign law/signal would attach here from the on-demand registry; the
        // builtin path needs nothing further.
        let _ = &mut builder;
        builder.build()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn window_parses_each_unit() {
        assert_eq!(
            Window::from_str("150ms").unwrap().0,
            Duration::from_millis(150)
        );
        assert_eq!(Window::from_str("2s").unwrap().0, Duration::from_secs(2));
        assert_eq!(
            Window::from_str("500us").unwrap().0,
            Duration::from_micros(500)
        );
        assert_eq!(Window::from_str("1m").unwrap().0, Duration::from_secs(60));
        assert_eq!(
            Window::from_str("150").unwrap().0,
            Duration::from_millis(150)
        );
        assert!(Window::from_str("3 fortnights").is_err());
    }

    #[test]
    fn window_round_trips_through_serde() {
        let window = Window(Duration::from_millis(150));
        let toml_text = toml::to_string(&Wrapper { window }).unwrap();
        assert!(toml_text.contains("150ms"), "got {toml_text}");
        let back: Wrapper = toml::from_str(&toml_text).unwrap();
        assert_eq!(back.window, window);
    }

    #[derive(Serialize, Deserialize)]
    struct Wrapper {
        window: Window,
    }

    #[test]
    fn defaults_resolve_to_gradient() {
        let settings = ConcurrencySettings::default();
        assert_eq!(settings.to_preset(), Preset::Gradient);
        assert_eq!(
            settings.to_bounds(),
            Bounds {
                min: 1,
                max: 512,
                start: 16
            }
        );
        assert!(Concurrency::from_settings(&settings).is_ok());
    }

    #[test]
    fn fixed_preset_resolves_to_fixed() {
        let settings = ConcurrencySettings {
            preset: PresetName::Fixed,
            fixed: 25,
            ..ConcurrencySettings::default()
        };
        let concurrency = Concurrency::from_settings(&settings).unwrap();
        assert!(matches!(concurrency, Concurrency::Fixed(25)));
    }

    use super::super::StrategyDescriptor;

    fn descriptor(concurrency: Concurrency) -> StrategyDescriptor {
        match concurrency {
            Concurrency::Adaptive(strategy) => strategy.descriptor(),
            Concurrency::Fixed(_) => panic!("expected adaptive"),
        }
    }

    // P4 parity: config (TOML), typed env, and the fluent builder must resolve to
    // the SAME strategy. Compared via the builtin descriptor — neither path is a
    // second source of truth that can drift.
    #[test]
    fn config_toml_and_fluent_paths_agree() {
        let via_toml: ConcurrencySettings = toml::from_str(
            "preset = \"gradient\"\n\
             coefficient_of_variation_threshold = 0.05\n\
             window = \"150ms\"\n\
             bounds = [1, 512]\n\
             start = 16\n",
        )
        .unwrap();
        let from_config = descriptor(Concurrency::from_settings(&via_toml).unwrap());

        let fluent = descriptor(
            Concurrency::builder()
                .gradient()
                .coefficient_of_variation_threshold(0.05)
                .window(Duration::from_millis(150))
                .bounds(1, 512)
                .start(16)
                .build()
                .unwrap(),
        );
        assert_eq!(from_config, fluent, "TOML config == fluent builder");
    }

    #[test]
    fn typed_env_matches_toml() {
        let via_toml: ConcurrencySettings = toml::from_str(
            "preset = \"hillclimb\"\ncoefficient_of_variation_threshold = 0.1\nwindow = \"200ms\"\n",
        )
        .unwrap();
        temp_env::with_vars(
            [
                ("CONCURRENCY_PRESET", Some("hillclimb")),
                (
                    "CONCURRENCY_COEFFICIENT_OF_VARIATION_THRESHOLD",
                    Some("0.1"),
                ),
                ("CONCURRENCY_WINDOW", Some("200ms")),
            ],
            || {
                let via_env = ConcurrencySettings::from_env().expect("env load");
                assert_eq!(via_env.preset, via_toml.preset);
                assert_eq!(
                    via_env.coefficient_of_variation_threshold,
                    via_toml.coefficient_of_variation_threshold
                );
                assert_eq!(via_env.window, via_toml.window);
                assert_eq!(
                    descriptor(Concurrency::from_settings(&via_env).unwrap()),
                    descriptor(Concurrency::from_settings(&via_toml).unwrap()),
                    "typed env == TOML"
                );
            },
        );
    }

    // preset path == fluent preset: from_preset and the sugar method agree.
    #[test]
    fn preset_path_matches_fluent_preset() {
        let from_preset = descriptor(Concurrency::from_preset(Preset::Gradient).unwrap());
        let fluent = descriptor(Concurrency::builder().gradient().build().unwrap());
        assert_eq!(from_preset, fluent);
    }

    // built -> config round-trip: a settings value serialises and reloads to the
    // same resolved strategy.
    #[test]
    fn settings_round_trip_through_toml() {
        let original = ConcurrencySettings {
            preset: PresetName::LatencyTarget,
            target_ms: 7.5,
            coefficient_of_variation_threshold: 0.08,
            ..ConcurrencySettings::default()
        };
        let serialized = toml::to_string(&original).unwrap();
        let reloaded: ConcurrencySettings = toml::from_str(&serialized).unwrap();
        assert_eq!(reloaded, original);
        assert_eq!(
            descriptor(Concurrency::from_settings(&reloaded).unwrap()),
            descriptor(Concurrency::from_settings(&original).unwrap()),
        );
    }

    // gate point 4: the resolver rejects an out-of-range bounds/start.
    #[test]
    fn settings_validate_rejects_bad_bounds() {
        let settings = ConcurrencySettings {
            bounds: BoundsPair([8, 4]),
            ..ConcurrencySettings::default()
        };
        assert!(settings.validate().is_err());
    }
}
