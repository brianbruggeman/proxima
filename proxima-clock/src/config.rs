//! Config + fluent-builder surface for [`AnchorCell`]'s initial `(ticks,
//! unix_nanos)` pair (guiding-principle 4). `conflaguration` is std-only
//! (see `rust.md`'s "Layering caveat" and
//! `proxima-telemetry/src/config.rs`, the canonical "one type = builder
//! result = config" pattern this mirrors), so this lives behind the
//! `config` feature at the std composition boundary — the no-alloc core
//! ([`crate::anchor`]) accepts a plain `(Ticks, UnixNanos)` pair in its
//! constructor and never sees this module.
//!
//! [`AnchorCell`] itself is not serializable (it is a live seqlock, not
//! data), so the config's "runtime form" bridge is explicit functions
//! ([`AnchorConfig::build`] / [`AnchorCell::to_config`]) rather than
//! [`AnchorCell`] holding an [`AnchorConfig`] field.

use bon::Builder;
use conflaguration::{Settings, Validate};
use serde::{Deserialize, Serialize};

use crate::anchor::AnchorCell;
use crate::ticks::Ticks;
use crate::unix_nanos::UnixNanos;

/// Serializable mirror of [`AnchorCell`]'s initial anchor pair.
///
/// `anchor_ticks` is the tick-domain half of the correlation point — for
/// most callers this is `0` (anchor to "whatever the counter reads right
/// now", read live at [`AnchorConfig::build`] time is NOT what happens:
/// this config is explicit data, so a `0` here means the counter's own
/// zero, not "read now"; a caller wanting "anchor to now" reads the
/// hardware counter once and passes that value in). `anchor_unix_nanos`
/// is the wall-clock correlation point: a PTP grandmaster's
/// `CLOCK_REALTIME` read at process start, or an NTP-disciplined
/// `SystemTime::now()` converted to nanoseconds since the Unix epoch.
#[derive(Debug, Clone, PartialEq, Eq, Builder, Serialize, Deserialize, Settings)]
#[settings(prefix = "PROXIMA_CLOCK")]
#[builder(derive(Clone, Debug))]
pub struct AnchorConfig {
    #[setting(default = 0)]
    #[serde(default)]
    #[builder(default)]
    pub anchor_ticks: u64,

    #[setting(default = 0)]
    #[serde(default)]
    #[builder(default)]
    pub anchor_unix_nanos: u64,
}

impl Default for AnchorConfig {
    fn default() -> Self {
        Self::builder().build()
    }
}

impl Validate for AnchorConfig {
    fn validate(&self) -> conflaguration::Result<()> {
        // both fields are plain u64 correlation points: every bit pattern
        // is a legal anchor (ticks=0/unix_nanos=0 is "anchored to the Unix
        // epoch at counter zero", a legitimate deterministic-test anchor,
        // not an error). nothing to reject.
        Ok(())
    }
}

impl AnchorConfig {
    /// Lower this config into a live [`AnchorCell`].
    #[must_use]
    pub fn build(&self) -> AnchorCell {
        AnchorCell::new(
            Ticks::from_raw(self.anchor_ticks),
            UnixNanos::from_nanos(self.anchor_unix_nanos),
        )
    }

    /// Load layered config from a file, then environment overrides
    /// (`PROXIMA_CLOCK_ANCHOR_TICKS` / `PROXIMA_CLOCK_ANCHOR_UNIX_NANOS`),
    /// falling back to defaults for anything neither source sets.
    ///
    /// # Errors
    ///
    /// Returns an error when the file cannot be read/parsed or an
    /// environment override fails to parse as `u64`.
    pub fn from_path_then_env<Path: AsRef<std::path::Path>>(
        path: Path,
    ) -> conflaguration::Result<Self> {
        conflaguration::from_file_then_env(path.as_ref())
    }
}

impl AnchorCell {
    /// The inverse of [`AnchorConfig::build`] — the current anchor pair as
    /// a serializable, round-trippable config snapshot.
    #[must_use]
    pub fn to_config(&self) -> AnchorConfig {
        let (ticks, unix_nanos) = self.get();
        AnchorConfig {
            anchor_ticks: ticks.as_raw(),
            anchor_unix_nanos: unix_nanos.as_nanos(),
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::AnchorConfig;
    use crate::ticks::Ticks;
    use crate::unix_nanos::UnixNanos;
    use conflaguration::Settings;
    use std::io::Write;

    #[test]
    fn default_equals_builder_build() {
        assert_eq!(AnchorConfig::default(), AnchorConfig::builder().build());
    }

    #[test]
    fn builder_and_env_loader_agree_on_an_equivalent_config() {
        temp_env::with_vars(
            [
                ("PROXIMA_CLOCK_ANCHOR_TICKS", Some("24000000")),
                ("PROXIMA_CLOCK_ANCHOR_UNIX_NANOS", Some("1753500000000000000")),
            ],
            || {
                let from_env = AnchorConfig::from_env().expect("from_env");
                let from_builder = AnchorConfig::builder()
                    .anchor_ticks(24_000_000)
                    .anchor_unix_nanos(1_753_500_000_000_000_000)
                    .build();

                assert_eq!(from_env, from_builder);
            },
        );
    }

    #[test]
    fn from_path_then_env_layers_file_under_env() {
        let mut file = tempfile::Builder::new()
            .suffix(".toml")
            .tempfile()
            .expect("tempfile");
        write!(file, "anchor_ticks = 100\nanchor_unix_nanos = 200\n").expect("write toml");

        temp_env::with_var("PROXIMA_CLOCK_ANCHOR_TICKS", Some("999"), || {
            let loaded = AnchorConfig::from_path_then_env(file.path()).expect("load");

            assert_eq!(loaded.anchor_ticks, 999, "env overrides the file");
            assert_eq!(loaded.anchor_unix_nanos, 200, "file value passes through");
        });
    }

    #[test]
    fn build_and_to_config_round_trip() {
        let config = AnchorConfig::builder()
            .anchor_ticks(24_000_000)
            .anchor_unix_nanos(1_753_500_000_000_000_000)
            .build();

        let cell = config.build();

        assert_eq!(cell.to_config(), config);
    }

    #[test]
    fn build_lowers_into_a_cell_reading_the_configured_anchor() {
        let config = AnchorConfig::builder()
            .anchor_ticks(500)
            .anchor_unix_nanos(9_000)
            .build();

        let cell = config.build();

        assert_eq!(
            cell.get(),
            (Ticks::from_raw(500), UnixNanos::from_nanos(9_000))
        );
    }
}
