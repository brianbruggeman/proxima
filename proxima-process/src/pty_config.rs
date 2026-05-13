//! `PtyConfig` + `PtySizeConfig` — first-class config surface for
//! [`PtyCommandPipe`](super::pty_pipe::PtyCommandPipe), mirroring
//! [`CommandConfig`](super::command_config::CommandConfig) for the
//! plain-spawn path.
//!
//! Per principle 4 in the pty-tester guiding-principles doc, every
//! component a user constructs exposes BOTH a config surface
//! (serde, conflaguration, layered loader) AND a fluent builder
//! surface (`bon::Builder`). This module is the PTY half of the pair.
//!
//! # Composition
//!
//! `PtyConfig` flattens a [`CommandConfig`] (so its TOML / JSON
//! layout is "all CommandConfig fields at the top level, plus a
//! `[size]` block"), and adds the PTY-specific `size`. Lowering
//! via [`PtyConfig::into_pty_command_pipe`] produces a built
//! [`PtyCommandPipe`].
//!
//! # PTY-specific overrides
//!
//! `PtyCommandPipe`'s `Pipe::call` always sets
//! `SpawnOptions::controlling_tty = true` at spawn time — the
//! child must own its session. The inner `CommandConfig`'s
//! `controlling_tty` field is therefore IGNORED on the PTY path;
//! it stays settable for round-trip fidelity but doesn't affect
//! the spawn. Set it via the std-shape `Command::controlling_tty`
//! method on a non-PTY `CommandPipe` if you want session leader
//! behaviour without a PTY.

extern crate alloc;

use alloc::format;

use bon::Builder;
use conflaguration::{Settings, Validate, ValidationMessage};
use serde::{Deserialize, Serialize};

use proxima_primitives::pipe::ProximaError;
use proxima_primitives::pipe::alloc_tier;

use super::command_config::CommandConfig;
use super::protocol::{ChildRequest, ChildResponse};
use super::pty::PtySize;
use super::pty_pipe::PtyCommandPipe;

/// Serialisable PTY window size — `(rows, cols)` in character
/// cells. Mirrors [`PtySize`] but carries `serde` + `bon` derives
/// for config-loader friendliness.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Builder, Serialize, Deserialize)]
pub struct PtySizeConfig {
    /// Number of rows (lines). Default 24.
    #[builder(default = 24)]
    #[serde(default = "default_rows")]
    pub rows: u16,
    /// Number of columns. Default 80.
    #[builder(default = 80)]
    #[serde(default = "default_cols")]
    pub cols: u16,
}

fn default_rows() -> u16 {
    24
}
fn default_cols() -> u16 {
    80
}

impl Default for PtySizeConfig {
    fn default() -> Self {
        Self { rows: 24, cols: 80 }
    }
}

impl From<PtySizeConfig> for PtySize {
    fn from(value: PtySizeConfig) -> Self {
        PtySize {
            rows: value.rows,
            cols: value.cols,
        }
    }
}

impl From<PtySize> for PtySizeConfig {
    fn from(value: PtySize) -> Self {
        PtySizeConfig {
            rows: value.rows,
            cols: value.cols,
        }
    }
}

/// First-class PTY config — a [`CommandConfig`] plus PTY window
/// size. Layered through `bon::Builder` + serde + conflaguration
/// the same way `CommandConfig` is, so a single TOML / env-var
/// load produces both halves.
#[derive(Debug, Clone, Default, Builder, Deserialize, Serialize, Settings)]
#[settings(prefix = "PROXIMA_PROCESS_PTY")]
#[builder(derive(Clone, Debug))]
pub struct PtyConfig {
    /// Inner command config — program / args / env / stdio /
    /// dispatch / libc_shim / etc. `#[serde(flatten)]` so the
    /// TOML layout reads as a single flat config without a
    /// `[command]` sub-table.
    ///
    /// Note: `controlling_tty` on the inner command is IGNORED
    /// by the PTY path (the PTY spawn always sets it true). It
    /// stays settable for round-trip fidelity.
    #[serde(flatten)]
    #[setting(skip)]
    #[builder(default)]
    pub command: CommandConfig,

    /// PTY window size. Defaults to 24×80.
    #[setting(skip)]
    #[serde(default)]
    #[builder(default)]
    pub size: PtySizeConfig,
}

impl Validate for PtyConfig {
    fn validate(&self) -> conflaguration::Result<()> {
        let mut errors = alloc::vec::Vec::new();
        if let Err(err) = self.command.validate() {
            errors.push(ValidationMessage::new("command", format!("{err}")));
        }
        if self.size.rows == 0 {
            errors.push(ValidationMessage::new("size.rows", "must be > 0"));
        }
        if self.size.cols == 0 {
            errors.push(ValidationMessage::new("size.cols", "must be > 0"));
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(conflaguration::Error::Validation { errors })
        }
    }
}

impl PtyConfig {
    /// Materialise the entire config into a built
    /// [`PtyCommandPipe`] — program + args + env + stdio +
    /// dispatch + libc_shim from the inner command config + PTY
    /// size, all wired through the typestate builder.
    pub fn into_pty_command_pipe(
        self,
    ) -> Result<PtyCommandPipe<alloc_tier::PipeHandle<ChildRequest, ChildResponse>>, ProximaError> {
        self.validate()
            .map_err(|err| ProximaError::Body(format!("{err}")))?;
        let chain = self.command.dispatch.clone().into_dyn_chain();
        let size: PtySize = self.size.into();
        let descriptor = self.command.into_command()?;
        let pipe = PtyCommandPipe::builder()
            .command(descriptor)?
            .size(size)
            .dispatch(chain)
            .build();
        Ok(pipe)
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
    use crate::env::Env;

    #[test]
    fn default_pty_size_config_is_24_by_80() {
        let size = PtySizeConfig::default();
        assert_eq!(size.rows, 24);
        assert_eq!(size.cols, 80);
    }

    #[test]
    fn pty_size_config_round_trips_with_pty_size() {
        let original = PtySize {
            rows: 50,
            cols: 132,
        };
        let via_config: PtySizeConfig = original.into();
        let back: PtySize = via_config.into();
        assert_eq!(back, original);
    }

    #[test]
    fn pty_config_rejects_zero_dimensions() {
        let mut cfg = PtyConfig::default();
        cfg.command.program = "/bin/true".to_string();
        cfg.size.rows = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn into_pty_command_pipe_materialises_from_config() {
        let cfg = PtyConfig::builder()
            .command(
                CommandConfig::builder()
                    .program("/bin/echo".to_string())
                    .args(alloc::vec!["hi".to_string()])
                    .env(Env::from_iter([("LANG", "C")]))
                    .build(),
            )
            .size(PtySizeConfig::builder().rows(40).cols(120).build())
            .build();
        let _pipe = cfg.into_pty_command_pipe().expect("materialise");
        // We can't easily exercise the PtyCommandPipe here without
        // a real PTY allocator — round-trip presence is the gate.
    }

    #[test]
    fn pty_config_serde_round_trip_via_toml() {
        let cfg = PtyConfig::builder()
            .command(
                CommandConfig::builder()
                    .program("/bin/echo".to_string())
                    .args(alloc::vec!["hi".to_string()])
                    .build(),
            )
            .size(PtySizeConfig::builder().rows(40).cols(120).build())
            .build();
        let toml_text = toml::to_string(&cfg).expect("serialise");
        let restored: PtyConfig = toml::from_str(&toml_text).expect("deserialise");
        assert_eq!(restored.size.rows, 40);
        assert_eq!(restored.size.cols, 120);
        assert_eq!(restored.command.program, "/bin/echo");
    }
}
