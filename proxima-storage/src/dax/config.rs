//! Configuration surface for the pmem DAX facade.
//!
//! [`PersistMode`] is always available. The richer [`DaxConfig`] (a `bon` fluent
//! builder + serde config + `conflaguration` env-layered `Settings`) ships
//! whenever the `dax` feature is on, mirroring the canonical proxima config
//! shape (`proxima-telemetry`). The config-free `PmemCowStore::{create_at,
//! open_at}` primitives remain available for callers that build their own
//! path/slot-length/persist-mode outside `DaxConfig`.

/// How [`crate::dax::PmemCowStore`] makes a region durable on `persist`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PersistMode {
    /// Real persistent memory mapped via DAX: a cache-line flush + fence is
    /// durability; no `msync` is needed (the mapping is the device).
    Dax,
    /// A regular file-backed mapping (testing, or non-pmem hosts): stores live in
    /// the page cache until synced, so `persist` adds an `msync` after the flush.
    #[default]
    FileBacked,
}

mod settings {
    use std::path::PathBuf;

    use bon::Builder;
    use conflaguration::{Settings, Validate, ValidationMessage};
    use serde::{Deserialize, Serialize};

    use super::PersistMode;

    /// Where and how the store maps its region.
    #[derive(Debug, Clone, Builder, Deserialize, Serialize, Settings)]
    #[settings(prefix = "PMEM_DAX")]
    #[builder(derive(Clone, Debug))]
    pub struct DaxConfig {
        /// Path to the pmem device (`/dev/dax*`) or the backing file. Real-world
        /// example: `/dev/dax0.0` on real pmem, or `/var/lib/service/store.pmem` for
        /// a file-backed test region.
        #[setting(default = "")]
        #[serde(default)]
        pub path: String,

        /// Per-value slot length in bytes. Two slots of this size plus the 8-byte
        /// root make up the region.
        #[setting(default = 0)]
        #[serde(default)]
        pub slot_len: usize,

        /// Durability mode for `persist`.
        #[setting(skip)]
        #[serde(default)]
        #[builder(default)]
        pub persist_mode: PersistMode,
    }

    impl DaxConfig {
        /// The configured path as a [`PathBuf`].
        #[must_use]
        pub fn path_buf(&self) -> PathBuf {
            PathBuf::from(&self.path)
        }
    }

    impl Validate for DaxConfig {
        fn validate(&self) -> conflaguration::Result<()> {
            let mut errors = Vec::new();
            if self.path.is_empty() {
                errors.push(ValidationMessage::new("path", "must be set"));
            }
            if self.slot_len == 0 {
                errors.push(ValidationMessage::new("slot_len", "must be > 0"));
            }
            if errors.is_empty() {
                Ok(())
            } else {
                Err(conflaguration::Error::Validation { errors })
            }
        }
    }

    #[cfg(test)]
    mod tests {
        #![allow(clippy::unwrap_used, clippy::expect_used)]
        use super::*;

        // P4: the fluent builder and the serde/config surface describe the same value.
        #[test]
        fn builder_and_toml_round_trip_agree() {
            let built = DaxConfig::builder()
                .path("/var/lib/service/store.pmem".to_owned())
                .slot_len(64)
                .persist_mode(PersistMode::Dax)
                .build();

            let toml_text = toml::to_string(&built).expect("serialize");
            let loaded: DaxConfig = toml::from_str(&toml_text).expect("deserialize");

            assert_eq!(loaded.path, built.path);
            assert_eq!(loaded.slot_len, built.slot_len);
            assert_eq!(loaded.persist_mode, built.persist_mode);
        }

        #[test]
        fn persist_mode_defaults_to_file_backed() {
            let config = DaxConfig::builder()
                .path("/tmp/x".to_owned())
                .slot_len(8)
                .build();
            assert_eq!(config.persist_mode, PersistMode::FileBacked);
        }

        #[test]
        fn validate_rejects_empty_path_and_zero_slot_len() {
            let bad_path = DaxConfig::builder().path(String::new()).slot_len(8).build();
            assert!(bad_path.validate().is_err());

            let bad_len = DaxConfig::builder()
                .path("/tmp/x".to_owned())
                .slot_len(0)
                .build();
            assert!(bad_len.validate().is_err());

            let good = DaxConfig::builder()
                .path("/tmp/x".to_owned())
                .slot_len(8)
                .build();
            assert!(good.validate().is_ok());
        }
    }
}

pub use settings::DaxConfig;
