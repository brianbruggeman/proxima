//! Config + fluent-builder surface for a handshake's per-session parameters
//! (guiding-principle 4, disciplined-component gate point 12).
//!
//! `conflaguration` is std-only, so this lives behind the `config` feature at
//! the std composition boundary. The no-alloc core ([`crate::handshake`])
//! takes plain values in its constructors and never sees this module — the
//! same split `proxima-clock` draws between `AnchorCell` and `AnchorConfig`.
//!
//! # What is deliberately not here
//!
//! **The pre-shared key is not a config field**, and that is the point rather
//! than an omission. Gate point 12 asks for sensitive fields to be flagged
//! `#[setting(sensitive)]`; a key that is never in the config surface at all
//! cannot be logged by a config dump, cannot be serialised into a snapshot,
//! and cannot be set from an environment variable visible in `ps`. It stays a
//! constructor argument, resolved by whatever key provider the composition
//! root uses.
//!
//! **Sizing is not here either.** The replay window, the DRBG counter width,
//! and the maximum payload are *build-time* constants baked from
//! `proxima-centauri.toml` (principle 12, see [`crate::sized`]). Those are
//! per-target choices that must be fixed when the binary is built, not
//! runtime knobs — a no-alloc deployment sizes its buffers against them.
//!
//! What remains is genuinely runtime deployment policy: which side of the
//! handshake this process plays, and which SPI it announces.

// this module only exists under `config`, which implies `std`; the crate is
// `#![no_std]` so std's prelude macros must be named explicitly.
use std::vec;

use bon::Builder;
use conflaguration::{Settings, Validate};
use serde::{Deserialize, Serialize};

use crate::handshake::{Handshake, IkeSpi, Role};

/// Serializable mirror of a handshake's per-session parameters.
///
/// `initiator` selects the side: a daemon that dials out is configured as one,
/// a listener as the other, and that is a deployment decision rather than a
/// property of the protocol. `spi` is the security parameter index this side
/// announces — pinned in config where an operator needs a stable identifier
/// across restarts, and otherwise left at its default for the composition root
/// to fill from a real entropy source.
#[derive(Debug, Clone, PartialEq, Eq, Builder, Serialize, Deserialize, Settings)]
#[settings(prefix = "PROXIMA_CENTAURI")]
#[builder(derive(Clone, Debug))]
pub struct HandshakeConfig {
    /// Whether this side opens the exchange.
    #[setting(default = false)]
    #[serde(default)]
    #[builder(default)]
    pub initiator: bool,

    /// This side's SPI. Zero means "unset" — the composition root supplies a
    /// drawn value rather than announcing a fixed one.
    #[setting(default = 0)]
    #[serde(default)]
    #[builder(default)]
    pub spi: u64,
}

impl Default for HandshakeConfig {
    fn default() -> Self {
        Self::builder().build()
    }
}

impl Validate for HandshakeConfig {
    fn validate(&self) -> conflaguration::Result<()> {
        // Every bit pattern is a legal SPI, including zero — which is the
        // documented "unset, draw one" sentinel rather than an error — and a
        // bool cannot be out of range. Nothing to reject.
        Ok(())
    }
}

impl HandshakeConfig {
    /// The role this config selects.
    #[must_use]
    pub const fn role(&self) -> Role {
        if self.initiator {
            Role::Initiator
        } else {
            Role::Responder
        }
    }

    /// Lower this config into a live [`Handshake`].
    ///
    /// The PSK is a separate argument on purpose: it is the one input that
    /// must not travel through a config file or an environment variable.
    #[must_use]
    pub const fn build(&self, psk: [u8; 32]) -> Handshake {
        let spi = IkeSpi::new(self.spi);
        if self.initiator {
            Handshake::initiator(psk, spi)
        } else {
            Handshake::responder(psk, spi)
        }
    }

    /// Load layered config from a file, then environment overrides
    /// (`PROXIMA_CENTAURI_INITIATOR` / `PROXIMA_CENTAURI_SPI`), falling back
    /// to defaults for anything neither source sets.
    ///
    /// # Errors
    ///
    /// Returns an error when the file cannot be read or parsed, or an
    /// environment override fails to parse.
    pub fn from_path_then_env<Path: AsRef<std::path::Path>>(
        path: Path,
    ) -> conflaguration::Result<Self> {
        conflaguration::from_file_then_env(path.as_ref())
    }
}

impl Handshake {
    /// The inverse of [`HandshakeConfig::build`] — this handshake's parameters
    /// as a serializable snapshot. Never includes the PSK.
    #[must_use]
    pub const fn to_config(&self) -> HandshakeConfig {
        HandshakeConfig {
            initiator: matches!(self.role(), Role::Initiator),
            spi: self.announced_spi().as_raw(),
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::format;
    use std::io::Write;

    use conflaguration::Settings;

    use super::HandshakeConfig;
    use crate::handshake::Role;

    const PSK: [u8; 32] = [0xAB; 32];

    #[test]
    fn default_equals_builder_build() {
        assert_eq!(
            HandshakeConfig::default(),
            HandshakeConfig::builder().build()
        );
    }

    #[test]
    fn the_builder_and_the_config_produce_equivalent_handshakes() {
        // gate point 12's parity fixture: construct the component both ways
        // and assert equivalent state.
        let from_config = HandshakeConfig {
            initiator: true,
            spi: 0x0102_0304_0506_0708,
        };
        let from_builder = HandshakeConfig::builder()
            .initiator(true)
            .spi(0x0102_0304_0506_0708)
            .build();

        assert_eq!(from_config, from_builder);

        let configured = from_config.build(PSK);
        let built = from_builder.build(PSK);

        assert_eq!(configured.to_config(), built.to_config());
        assert_eq!(configured.role(), Role::Initiator);
        assert_eq!(configured.role(), built.role());
    }

    #[test]
    fn a_handshake_round_trips_through_its_config() {
        let original = HandshakeConfig::builder()
            .initiator(true)
            .spi(0xDEAD)
            .build();

        let round_tripped = original.build(PSK).to_config();

        assert_eq!(round_tripped, original, "build then to_config is identity");
    }

    #[test]
    fn the_responder_is_the_default_side() {
        let config = HandshakeConfig::default();

        assert_eq!(config.role(), Role::Responder);
        assert_eq!(config.build(PSK).role(), Role::Responder);
    }

    #[test]
    fn env_overrides_reach_the_built_handshake() {
        temp_env::with_vars(
            [
                ("PROXIMA_CENTAURI_INITIATOR", Some("true")),
                ("PROXIMA_CENTAURI_SPI", Some("4919")),
            ],
            || {
                let config = HandshakeConfig::from_env().expect("env parses");

                assert!(config.initiator);
                assert_eq!(config.spi, 4919);
                assert_eq!(config.build(PSK).role(), Role::Initiator);
            },
        );
    }

    #[test]
    fn a_file_supplies_defaults_that_env_then_overrides() {
        // conflaguration selects its parser from the extension, so the
        // suffix is load-bearing rather than cosmetic
        let mut file = tempfile::Builder::new()
            .suffix(".toml")
            .tempfile()
            .expect("temp file");
        writeln!(file, "initiator = true").expect("write");
        writeln!(file, "spi = 111").expect("write");

        temp_env::with_vars([("PROXIMA_CENTAURI_SPI", Some("222"))], || {
            let config =
                HandshakeConfig::from_path_then_env(file.path()).expect("file then env loads");

            assert!(config.initiator, "came from the file");
            assert_eq!(config.spi, 222, "environment wins over the file");
        });
    }

    #[test]
    fn the_config_surface_carries_no_key_material() {
        // the PSK is a constructor argument, never a field: a config dump, a
        // serialised snapshot, and an environment variable all cannot leak it.
        let rendered = format!("{:?}", HandshakeConfig::builder().spi(7).build());

        assert!(!rendered.contains("psk"), "no psk field: {rendered}");
        assert!(!rendered.contains("171"), "no 0xAB key bytes: {rendered}");

        let serialised = toml::to_string(&HandshakeConfig::default()).expect("serialises");
        assert!(!serialised.contains("psk"), "no psk in toml: {serialised}");
    }
}
