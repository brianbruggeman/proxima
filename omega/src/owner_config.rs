//! `OwnerConfig` -- the device owner's runtime sizing: config surface AND fluent builder.
//!
//! Every default is seeded from the `sized` consts `build.rs` generates from
//! `omega-runtime.toml`'s `[owner]` section, so the runtime default follows the
//! build-time floor with no second source of truth. `conflaguration`'s derive
//! only accepts literal defaults, so [`Settings`] is implemented by hand over
//! [`resolve_or`], the same call the derive expands to.
//!
//! [`OwnerConfig::layered`] returns a [`OwnerConfigLayerBuilder`] with
//! call-order precedence: `.with_*` before `.from_path`/`.from_env` lets operator
//! config win; after them, code wins. Consumed by
//! [`crate::device_owner::launch_on_thread`].

use std::path::Path;

use bon::Builder;
use conflaguration::{Settings, Validate, ValidationMessage, resolve_or};
use serde::{Deserialize, Serialize};

use crate::sized::{OWNER_OWNED_HOST_BYTES, OWNER_PLAN_CAPACITY, OWNER_QUEUE_DEPTH};

const QUEUE_DEPTH_ENV: &str = "OMEGA_DEVICE_OWNER_QUEUE_DEPTH";
const PLAN_CAPACITY_ENV: &str = "OMEGA_DEVICE_OWNER_PLAN_CAPACITY";
const OWNED_HOST_BYTES_ENV: &str = "OMEGA_DEVICE_OWNER_OWNED_HOST_BYTES";

fn default_queue_depth() -> usize {
    OWNER_QUEUE_DEPTH
}

fn default_plan_capacity() -> usize {
    OWNER_PLAN_CAPACITY
}

fn default_owned_host_bytes() -> usize {
    OWNER_OWNED_HOST_BYTES
}

/// Sizing for one device owner thread.
#[derive(Debug, Clone, PartialEq, Eq, Builder, Deserialize, Serialize)]
#[builder(derive(Clone, Debug))]
pub struct OwnerConfig {
    /// Depth of the bounded command queue; a full queue back-pressures callers.
    #[serde(default = "default_queue_depth")]
    #[builder(default = default_queue_depth())]
    pub queue_depth: usize,

    /// Registered plans the owner keeps resident at once.
    #[serde(default = "default_plan_capacity")]
    #[builder(default = default_plan_capacity())]
    pub plan_capacity: usize,

    /// Ceiling on host bytes the owner copies out of caller blocks.
    #[serde(default = "default_owned_host_bytes")]
    #[builder(default = default_owned_host_bytes())]
    pub owned_host_bytes: usize,
}

impl Default for OwnerConfig {
    fn default() -> Self {
        OwnerConfig::builder().build()
    }
}

impl Settings for OwnerConfig {
    const PREFIX: Option<&'static str> = Some("OMEGA_DEVICE_OWNER");

    fn from_env() -> conflaguration::Result<Self> {
        Ok(Self {
            queue_depth: resolve_or(&[QUEUE_DEPTH_ENV], OWNER_QUEUE_DEPTH)?,
            plan_capacity: resolve_or(&[PLAN_CAPACITY_ENV], OWNER_PLAN_CAPACITY)?,
            owned_host_bytes: resolve_or(&[OWNED_HOST_BYTES_ENV], OWNER_OWNED_HOST_BYTES)?,
        })
    }
}

impl Validate for OwnerConfig {
    fn validate(&self) -> conflaguration::Result<()> {
        let errors: Vec<ValidationMessage> = [
            ("queue_depth", self.queue_depth),
            ("plan_capacity", self.plan_capacity),
            ("owned_host_bytes", self.owned_host_bytes),
        ]
        .into_iter()
        .filter(|(_, value)| *value == 0)
        .map(|(field, _)| ValidationMessage::new(field, "must be > 0"))
        .collect();
        if errors.is_empty() {
            Ok(())
        } else {
            Err(conflaguration::Error::Validation { errors })
        }
    }
}

impl OwnerConfig {
    /// Start a layered builder from the `sized`-seeded defaults.
    #[must_use]
    pub fn layered() -> OwnerConfigLayerBuilder {
        OwnerConfigLayerBuilder {
            inner: OwnerConfig::default(),
        }
    }
}

#[derive(Debug, Default, Deserialize)]
struct OwnerConfigPartial {
    queue_depth: Option<usize>,
    plan_capacity: Option<usize>,
    owned_host_bytes: Option<usize>,
}

/// Fluent builder for [`OwnerConfig`]: each source contributes only the fields
/// it specifies, last writer wins per field.
pub struct OwnerConfigLayerBuilder {
    inner: OwnerConfig,
}

impl OwnerConfigLayerBuilder {
    /// Merge a TOML/JSON file's fields onto the accumulated config.
    pub fn from_path<P: AsRef<Path>>(mut self, path: P) -> Result<Self, conflaguration::Error> {
        let partial: OwnerConfigPartial = conflaguration::from_file(path.as_ref())?;
        self.inner.queue_depth = partial.queue_depth.unwrap_or(self.inner.queue_depth);
        self.inner.plan_capacity = partial.plan_capacity.unwrap_or(self.inner.plan_capacity);
        self.inner.owned_host_bytes = partial
            .owned_host_bytes
            .unwrap_or(self.inner.owned_host_bytes);
        Ok(self)
    }

    /// Merge env-set fields onto the accumulated config; unset vars leave the
    /// current value untouched.
    pub fn from_env(mut self) -> Result<Self, conflaguration::Error> {
        self.inner.queue_depth = resolve_or(&[QUEUE_DEPTH_ENV], self.inner.queue_depth)?;
        self.inner.plan_capacity = resolve_or(&[PLAN_CAPACITY_ENV], self.inner.plan_capacity)?;
        self.inner.owned_host_bytes =
            resolve_or(&[OWNED_HOST_BYTES_ENV], self.inner.owned_host_bytes)?;
        Ok(self)
    }

    /// Set the command queue depth.
    #[must_use]
    pub fn with_queue_depth(mut self, queue_depth: usize) -> Self {
        self.inner.queue_depth = queue_depth;
        self
    }

    /// Set the resident plan capacity.
    #[must_use]
    pub fn with_plan_capacity(mut self, plan_capacity: usize) -> Self {
        self.inner.plan_capacity = plan_capacity;
        self
    }

    /// Set the owned host byte ceiling.
    #[must_use]
    pub fn with_owned_host_bytes(mut self, owned_host_bytes: usize) -> Self {
        self.inner.owned_host_bytes = owned_host_bytes;
        self
    }

    /// The built immutable config.
    #[must_use]
    pub fn build(self) -> OwnerConfig {
        self.inner
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    const UNSET: [(&str, Option<&str>); 3] = [
        (QUEUE_DEPTH_ENV, None),
        (PLAN_CAPACITY_ENV, None),
        (OWNED_HOST_BYTES_ENV, None),
    ];

    #[test]
    fn defaults_track_the_sized_floor() {
        let config = OwnerConfig::default();
        assert_eq!(config.queue_depth, OWNER_QUEUE_DEPTH);
        assert_eq!(config.plan_capacity, OWNER_PLAN_CAPACITY);
        assert_eq!(config.owned_host_bytes, OWNER_OWNED_HOST_BYTES);
        temp_env::with_vars(UNSET, || {
            let from_env = OwnerConfig::from_env().expect("from_env with nothing set");
            assert_eq!(from_env, config);
        });
    }

    #[test]
    fn env_and_fluent_build_the_same_owner_config() {
        let fluent = OwnerConfig::builder()
            .queue_depth(7)
            .plan_capacity(9)
            .owned_host_bytes(4096)
            .build();
        temp_env::with_vars(
            [
                (QUEUE_DEPTH_ENV, Some("7")),
                (PLAN_CAPACITY_ENV, Some("9")),
                (OWNED_HOST_BYTES_ENV, Some("4096")),
            ],
            || {
                let from_env = OwnerConfig::from_env().expect("from_env");
                let layered = OwnerConfig::layered()
                    .from_env()
                    .expect("layered env")
                    .build();
                assert_eq!(from_env, fluent);
                assert_eq!(layered, fluent);
            },
        );
    }

    #[test]
    fn a_zero_queue_depth_fails_validation() {
        let config = OwnerConfig::layered().with_queue_depth(0).build();
        let error = config
            .validate()
            .expect_err("zero queue depth must be rejected");
        assert!(format!("{error:?}").contains("queue_depth"));
    }
}
