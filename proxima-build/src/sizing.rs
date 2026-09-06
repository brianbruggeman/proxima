//! Shared build-time sizing-constant resolution (guiding-principles §12):
//! reads a per-crate `<crate>-runtime.toml`, resolves `[section].key`
//! values with a per-key `<ENV_PREFIX>_<SECTION>_<KEY>` env override, and
//! emits the `cargo:rerun-if-changed` / `cargo:rerun-if-env-changed`
//! directives every consulted source needs.
//!
//! Typical consumer (`build.rs`):
//!
//! ```no_run
//! # fn main() -> proxima_build::sizing::Result<()> {
//! let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").expect("set by cargo");
//! let source = proxima_build::sizing::SizingSource::load(
//!     &manifest_dir,
//!     "omega-runtime.toml",
//!     "OMEGA",
//! )?;
//! let simdgroups = proxima_build::sizing::require_nonzero(
//!     "packed_row_block.simdgroups",
//!     source.resolve_int("packed_row_block", "simdgroups")?,
//! );
//! # Ok(())
//! # }
//! ```
//!
//! Cross-axis validation (a value must be a power of two, a multiple of a
//! kernel's tile width, and so on) stays with the consuming crate's own
//! `build.rs` — those rules encode that crate's kernel geometry, not a
//! property of sizing resolution itself.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use toml::Value;

/// Errors resolving a numeric build-time sizing constant.
#[derive(Debug, thiserror::Error)]
pub enum SizingError {
    #[error("read {0}: {1}")]
    Read(PathBuf, std::io::Error),
    #[error("parse {0}: {1}")]
    Parse(PathBuf, toml::de::Error),
    #[error("{toml}: missing or non-integer [{section}].{key}", toml = .toml.display())]
    MissingInt {
        toml: PathBuf,
        section: String,
        key: String,
    },
    #[error("{toml}: missing or non-float [{section}].{key}", toml = .toml.display())]
    MissingFloat {
        toml: PathBuf,
        section: String,
        key: String,
    },
    #[error("{toml}: missing or non-string [{section}].{key}", toml = .toml.display())]
    MissingStr {
        toml: PathBuf,
        section: String,
        key: String,
    },
    #[error("{toml}: missing or non-bool [{section}].{key}", toml = .toml.display())]
    MissingBool {
        toml: PathBuf,
        section: String,
        key: String,
    },
    #[error("{env_name}={raw} must parse as bool: {source}")]
    EnvBool {
        env_name: String,
        raw: String,
        source: std::str::ParseBoolError,
    },
    #[error("{env_name}={raw} must parse as i64: {source}")]
    EnvInt {
        env_name: String,
        raw: String,
        source: std::num::ParseIntError,
    },
    #[error("{env_name}={raw} must parse as f64: {source}")]
    EnvFloat {
        env_name: String,
        raw: String,
        source: std::num::ParseFloatError,
    },
}

pub type Result<T> = core::result::Result<T, SizingError>;

/// A loaded `<crate>-runtime.toml`, ready to resolve individual
/// `[section].key` values against `<ENV_PREFIX>_<SECTION>_<KEY>` overrides.
pub struct SizingSource {
    toml_path: PathBuf,
    table: Value,
    env_prefix: String,
}

impl SizingSource {
    /// Reads `<manifest_dir>/<toml_filename>` and emits its
    /// `cargo:rerun-if-changed` directive.
    ///
    /// # Errors
    ///
    /// [`SizingError::Read`] if the file cannot be read;
    /// [`SizingError::Parse`] if it is not valid TOML.
    pub fn load(manifest_dir: &str, toml_filename: &str, env_prefix: &str) -> Result<Self> {
        let toml_path = Path::new(manifest_dir).join(toml_filename);
        println!("cargo:rerun-if-changed={}", toml_path.display());
        let text =
            fs::read_to_string(&toml_path).map_err(|err| SizingError::Read(toml_path.clone(), err))?;
        let table: Value =
            text.parse().map_err(|err| SizingError::Parse(toml_path.clone(), err))?;
        Ok(Self {
            toml_path,
            table,
            env_prefix: env_prefix.to_owned(),
        })
    }

    fn env_name(&self, section: &str, key: &str) -> String {
        format!(
            "{prefix}_{section}_{key}",
            prefix = self.env_prefix,
            section = section.to_uppercase(),
            key = key.to_uppercase()
        )
    }

    /// Resolves `[section].key` as an integer, honoring an
    /// `<ENV_PREFIX>_<SECTION>_<KEY>` env override. Always emits the
    /// override's `cargo:rerun-if-env-changed` directive, so a cached build
    /// never ignores a later export of that variable.
    ///
    /// # Errors
    ///
    /// [`SizingError::EnvInt`] if the override is set but does not parse;
    /// [`SizingError::MissingInt`] if neither the override nor the TOML key
    /// is present.
    pub fn resolve_int(&self, section: &str, key: &str) -> Result<i64> {
        let env_name = self.env_name(section, key);
        println!("cargo:rerun-if-env-changed={env_name}");
        match env::var(&env_name) {
            Ok(raw) => raw
                .parse::<i64>()
                .map_err(|source| SizingError::EnvInt { env_name, raw, source }),
            Err(_) => self
                .table
                .get(section)
                .and_then(|section_value| section_value.get(key))
                .and_then(Value::as_integer)
                .ok_or_else(|| SizingError::MissingInt {
                    toml: self.toml_path.clone(),
                    section: section.to_owned(),
                    key: key.to_owned(),
                }),
        }
    }

    /// Like [`Self::resolve_int`], but for a floating-point key.
    ///
    /// # Errors
    ///
    /// [`SizingError::EnvFloat`] if the override is set but does not parse;
    /// [`SizingError::MissingFloat`] if neither the override nor the TOML
    /// key is present.
    pub fn resolve_float(&self, section: &str, key: &str) -> Result<f64> {
        let env_name = self.env_name(section, key);
        println!("cargo:rerun-if-env-changed={env_name}");
        match env::var(&env_name) {
            Ok(raw) => raw
                .parse::<f64>()
                .map_err(|source| SizingError::EnvFloat { env_name, raw, source }),
            Err(_) => self
                .table
                .get(section)
                .and_then(|section_value| section_value.get(key))
                .and_then(Value::as_float)
                .ok_or_else(|| SizingError::MissingFloat {
                    toml: self.toml_path.clone(),
                    section: section.to_owned(),
                    key: key.to_owned(),
                }),
        }
    }

    /// Like [`Self::resolve_int`], but for a string key. Returns an owned
    /// `String` because the override value (when present) only lives as
    /// long as the `env::var` call, so a borrowed form can't outlive it.
    ///
    /// # Errors
    ///
    /// [`SizingError::MissingStr`] if neither the override nor the TOML key
    /// is present.
    pub fn resolve_str(&self, section: &str, key: &str) -> Result<String> {
        let env_name = self.env_name(section, key);
        println!("cargo:rerun-if-env-changed={env_name}");
        match env::var(&env_name) {
            Ok(raw) => Ok(raw),
            Err(_) => self
                .table
                .get(section)
                .and_then(|section_value| section_value.get(key))
                .and_then(Value::as_str)
                .map(str::to_owned)
                .ok_or_else(|| SizingError::MissingStr {
                    toml: self.toml_path.clone(),
                    section: section.to_owned(),
                    key: key.to_owned(),
                }),
        }
    }

    /// Like [`Self::resolve_int`], but for a boolean key.
    ///
    /// # Errors
    ///
    /// [`SizingError::EnvBool`] if the override is set but does not parse;
    /// [`SizingError::MissingBool`] if neither the override nor the TOML key
    /// is present.
    pub fn resolve_bool(&self, section: &str, key: &str) -> Result<bool> {
        let env_name = self.env_name(section, key);
        println!("cargo:rerun-if-env-changed={env_name}");
        match env::var(&env_name) {
            Ok(raw) => raw
                .parse::<bool>()
                .map_err(|source| SizingError::EnvBool { env_name, raw, source }),
            Err(_) => self
                .table
                .get(section)
                .and_then(|section_value| section_value.get(key))
                .and_then(Value::as_bool)
                .ok_or_else(|| SizingError::MissingBool {
                    toml: self.toml_path.clone(),
                    section: section.to_owned(),
                    key: key.to_owned(),
                }),
        }
    }
}

/// Requires `value` to be a positive integer, returning it as `usize`.
/// Panics with a build.rs-appropriate message otherwise — the shared shape
/// every consumer's own `require_*` cross-axis rule builds on.
#[allow(clippy::expect_used)]
pub fn require_nonzero(name: &str, value: i64) -> usize {
    let value = usize::try_from(value)
        .unwrap_or_else(|_| panic!("{name} must be a non-negative integer; got {value}"));
    assert!(value > 0, "{name} must be non-zero");
    value
}

/// Like [`require_nonzero`], but `0` is a legal, documented sentinel value.
#[allow(clippy::expect_used)]
pub fn require_nonneg(name: &str, value: i64) -> usize {
    usize::try_from(value)
        .unwrap_or_else(|_| panic!("{name} must be a non-negative integer; got {value}"))
}

#[cfg(test)]
// workspace lints deny unwrap/expect; tests are the sanctioned exception.
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn write_temp_toml(dir: &Path, contents: &str) -> PathBuf {
        let path = dir.join("crate-runtime.toml");
        fs::write(&path, contents).unwrap();
        path
    }

    #[test]
    fn resolve_int_reads_toml_value_when_no_env_override() {
        let tmp = tempfile::tempdir().unwrap();
        write_temp_toml(tmp.path(), "[section]\nkey = 7\n");
        temp_env::with_var("TEST_SECTION_KEY", None::<&str>, || {
            let source =
                SizingSource::load(tmp.path().to_str().unwrap(), "crate-runtime.toml", "TEST")
                    .expect("load");
            assert_eq!(source.resolve_int("section", "key").unwrap(), 7);
        });
    }

    #[test]
    fn resolve_int_env_override_wins_over_toml() {
        let tmp = tempfile::tempdir().unwrap();
        write_temp_toml(tmp.path(), "[section]\nkey = 7\n");
        temp_env::with_var("TEST_SECTION_KEY", Some("42"), || {
            let source =
                SizingSource::load(tmp.path().to_str().unwrap(), "crate-runtime.toml", "TEST")
                    .expect("load");
            assert_eq!(source.resolve_int("section", "key").unwrap(), 42);
        });
    }

    #[test]
    fn resolve_int_missing_key_is_typed_error() {
        let tmp = tempfile::tempdir().unwrap();
        write_temp_toml(tmp.path(), "[section]\nother = 1\n");
        temp_env::with_var("TEST_SECTION_KEY", None::<&str>, || {
            let source =
                SizingSource::load(tmp.path().to_str().unwrap(), "crate-runtime.toml", "TEST")
                    .expect("load");
            let err = source.resolve_int("section", "key").expect_err("missing key");
            assert!(matches!(err, SizingError::MissingInt { .. }));
        });
    }

    #[test]
    fn resolve_float_reads_toml_value_when_no_env_override() {
        let tmp = tempfile::tempdir().unwrap();
        write_temp_toml(tmp.path(), "[section]\nkey = 1.5\n");
        temp_env::with_var("TEST_SECTION_KEY", None::<&str>, || {
            let source =
                SizingSource::load(tmp.path().to_str().unwrap(), "crate-runtime.toml", "TEST")
                    .expect("load");
            assert!((source.resolve_float("section", "key").unwrap() - 1.5).abs() < f64::EPSILON);
        });
    }

    #[test]
    fn require_nonzero_accepts_positive_value() {
        assert_eq!(require_nonzero("axis", 3), 3);
    }

    #[test]
    #[should_panic(expected = "axis must be non-zero")]
    fn require_nonzero_rejects_zero() {
        require_nonzero("axis", 0);
    }

    #[test]
    fn require_nonneg_accepts_zero_sentinel() {
        assert_eq!(require_nonneg("axis", 0), 0);
    }
}
