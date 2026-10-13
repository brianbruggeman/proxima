//! Configuration for the on-disk CUDA PTX cache, loaded through conflaguration.
//!
//! The derive owns the default through `default_str`: an unset
//! `PROXIMA_CUDA_KERNEL_CACHE_DIR` resolves to `~/.cache/proxima/cuda-ptx`, and
//! the parse fn expands the leading `~/` against `$HOME`. Use the `Settings`
//! trait path to load; there is no second loader.

use conflaguration::{Settings, Validate};
use serde::{Deserialize, Deserializer, Serialize};
use std::convert::Infallible;
use std::path::PathBuf;

const DEFAULT_KERNEL_CACHE_DIR: &str = "~/.cache/proxima/cuda-ptx";

/// Where compiled CUDA PTX is kept between processes.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, Settings, Validate)]
#[settings(prefix = "PROXIMA_CUDA")]
#[serde(default, deny_unknown_fields)]
pub struct KernelCacheConfig {
    /// Directory holding one PTX file per kernel cache key, such as
    /// `/home/bix/.cache/proxima/cuda-ptx`. `None` keeps kernels in memory only;
    /// the env spelling of `None` is an empty value.
    #[setting(
        resolve_with = "parse_kernel_cache_dir",
        default_str = "~/.cache/proxima/cuda-ptx"
    )]
    #[serde(deserialize_with = "deserialize_kernel_cache_dir")]
    pub kernel_cache_dir: Option<PathBuf>,
}

impl Default for KernelCacheConfig {
    fn default() -> Self {
        Self {
            kernel_cache_dir: parse_kernel_cache_dir(DEFAULT_KERNEL_CACHE_DIR).unwrap_or(None),
        }
    }
}

impl KernelCacheConfig {
    /// Replaces the cache directory; `None` disables the on-disk cache.
    #[must_use]
    pub fn with_kernel_cache_dir(mut self, value: Option<PathBuf>) -> Self {
        self.kernel_cache_dir = value;
        self
    }
}

fn parse_kernel_cache_dir(value: &str) -> Result<Option<PathBuf>, Infallible> {
    if value.is_empty() {
        return Ok(None);
    }
    let Some(relative) = value.strip_prefix("~/") else {
        return Ok(Some(PathBuf::from(value)));
    };
    Ok(home_relative(relative))
}

fn home_relative(relative: &str) -> Option<PathBuf> {
    let Some(home) = std::env::var_os("HOME") else {
        proxima_telemetry::warn!(
            "HOME is unset; kernel disk cache disabled until PROXIMA_CUDA_KERNEL_CACHE_DIR is set"
        );
        return None;
    };
    Some(PathBuf::from(home).join(relative))
}

fn deserialize_kernel_cache_dir<'de, D>(deserializer: D) -> Result<Option<PathBuf>, D::Error>
where
    D: Deserializer<'de>,
{
    Ok(Option::<PathBuf>::deserialize(deserializer)?.filter(|dir| !dir.as_os_str().is_empty()))
}

#[cfg(test)]
mod tests {
    use super::KernelCacheConfig;
    use conflaguration::Settings;
    use std::path::PathBuf;

    const CACHE_VAR: &str = "PROXIMA_CUDA_KERNEL_CACHE_DIR";

    #[test]
    fn derive_path_unset_gives_home_cache_default() {
        let home = tempfile::tempdir().expect("tempdir for home");
        temp_env::with_vars(
            [
                (CACHE_VAR, None::<&str>),
                ("HOME", Some(home.path().to_str().expect("utf8 home"))),
            ],
            || {
                let config = KernelCacheConfig::from_env().expect("unset env loads defaults");
                assert_eq!(
                    config.kernel_cache_dir,
                    Some(home.path().join(".cache/proxima/cuda-ptx"))
                );
            },
        );
    }

    #[test]
    fn derive_path_without_home_is_disabled() {
        temp_env::with_vars([(CACHE_VAR, None::<&str>), ("HOME", None::<&str>)], || {
            let config = KernelCacheConfig::from_env().expect("unset env loads defaults");
            assert_eq!(config.kernel_cache_dir, None);
        });
    }

    #[test]
    fn derive_path_empty_env_gives_none() {
        temp_env::with_vars([(CACHE_VAR, Some(""))], || {
            let config = KernelCacheConfig::from_env().expect("empty env loads");
            assert_eq!(config.kernel_cache_dir, None);
        });
    }

    #[test]
    fn derive_path_tilde_env_expands_under_home() {
        let home = tempfile::tempdir().expect("tempdir for home");
        temp_env::with_vars(
            [
                (CACHE_VAR, Some("~/custom/ptx")),
                ("HOME", Some(home.path().to_str().expect("utf8 home"))),
            ],
            || {
                let config = KernelCacheConfig::from_env().expect("tilde env loads");
                assert_eq!(
                    config.kernel_cache_dir,
                    Some(home.path().join("custom/ptx"))
                );
            },
        );
    }

    #[test]
    fn kernel_cache_config_builder_env_and_toml_are_equal() {
        let dir = tempfile::tempdir().expect("tempdir for cache root");
        let cache_path = dir.path().join("cuda-ptx");
        let toml_path = dir.path().join("kernel_cache.toml");
        let toml_body = format!(
            "kernel_cache_dir = {:?}\n",
            cache_path.display().to_string()
        );
        std::fs::write(&toml_path, toml_body).expect("write kernel cache toml");

        let fluent = KernelCacheConfig::default().with_kernel_cache_dir(Some(cache_path.clone()));
        let from_env = temp_env::with_vars([(CACHE_VAR, Some(cache_path.as_os_str()))], || {
            KernelCacheConfig::from_env().expect("env loads")
        });
        let from_toml = conflaguration::builder()
            .value(KernelCacheConfig::default())
            .file(&toml_path)
            .validate()
            .build()
            .expect("toml loads");

        assert_eq!(fluent, from_env, "fluent builder matches env");
        assert_eq!(fluent, from_toml, "fluent builder matches toml");
    }

    #[test]
    fn kernel_cache_config_toml_empty_string_disables_the_cache() {
        let dir = tempfile::tempdir().expect("tempdir for toml");
        let toml_path = dir.path().join("kernel_cache.toml");
        std::fs::write(&toml_path, "kernel_cache_dir = \"\"\n").expect("write empty toml");

        let config = conflaguration::builder()
            .value(KernelCacheConfig::default())
            .file(&toml_path)
            .validate()
            .build()
            .expect("toml loads");

        assert_eq!(config.kernel_cache_dir, None);
    }

    #[test]
    fn kernel_cache_config_disabled_builder_value_is_none() {
        let config = KernelCacheConfig::default().with_kernel_cache_dir(None);
        assert_eq!(config.kernel_cache_dir, None::<PathBuf>);
    }
}
