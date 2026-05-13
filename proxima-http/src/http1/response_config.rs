//! Response-handling composition knobs carried on every HTTP/1.1 upstream.
//!
//! These are pure config types (serde + conflaguration + builder, no I/O),
//! so they live OUTSIDE the `stream-client`-gated `client` module: the
//! always-compiled `HttpUpstreamConfig` (in `upstream`) carries a
//! `ResponseHandlingConfig` field, so the type must exist even when the
//! prime-native client is not built. The client module and the prime
//! upstream re-use these same types.

use bon::Builder;
use conflaguration::{Settings, Validate};
use serde::{Deserialize, Serialize};

/// What the client does with response body bytes once the framing decoder has
/// established the message boundary.
///
/// `Collect` (default) materializes the full body into `Bytes` — the
/// correct choice for any proxy or caller that reads the body.
///
/// `Drain` reads every frame to find the keep-alive boundary then discards
/// the payload — zero `Bytes` allocation or copy. The right choice for load
/// generators and health probers that only need the status code.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ResponseBodyMode {
    #[default]
    Collect,
    Drain,
}

/// Which response headers the client copies into the returned `Response`.
///
/// `All` (default) copies every header into a full `HeaderList` —
/// exactly the existing behaviour.
///
/// `Framing` only copies the three headers the client itself needs to
/// advance the connection state (content-length, transfer-encoding,
/// connection). Everything else is skipped — no `Bytes::copy_from_slice`
/// per header, no `HeaderList` entry allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ResponseHeaderMode {
    #[default]
    All,
    Framing,
}

/// A named preset that composes [`ResponseBodyMode`] + [`ResponseHeaderMode`]
/// for the common cases. Granular control via [`ResponseHandlingConfig`] is
/// always available and takes precedence.
///
/// `Full` (default) = `Collect` + `All` — the unchanged generic path.
/// `Discard` = `Drain` + `Framing` — load-gen / health-probe composition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ResponseHandling {
    Full,
    Discard,
}

/// Composed response-handling config carried on every `H1ClientUpstream`.
/// Decided once at client build; the hot path sees a cheap integer comparison,
/// not a per-request decision.
///
/// TOML surface (under `[client.response]`):
/// ```toml
/// [client.response]
/// body    = "collect"   # default
/// headers = "all"       # default
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Builder, Deserialize, Serialize, Settings)]
#[settings(prefix = "H1_CLIENT_RESPONSE")]
#[builder(derive(Clone, Debug))]
pub struct ResponseHandlingConfig {
    /// What to do with response body bytes.
    #[setting(skip)]
    #[serde(default)]
    #[builder(default)]
    pub body: ResponseBodyMode,

    /// Which response headers to retain.
    #[setting(skip)]
    #[serde(default)]
    #[builder(default)]
    pub headers: ResponseHeaderMode,
}

impl Default for ResponseHandlingConfig {
    fn default() -> Self {
        Self::builder().build()
    }
}

impl ResponseHandlingConfig {
    /// Apply a named preset, overriding both axes.
    #[must_use]
    pub fn from_preset(preset: ResponseHandling) -> Self {
        match preset {
            ResponseHandling::Full => Self::default(),
            ResponseHandling::Discard => Self {
                body: ResponseBodyMode::Drain,
                headers: ResponseHeaderMode::Framing,
            },
        }
    }
}

impl Validate for ResponseHandlingConfig {
    fn validate(&self) -> conflaguration::Result<()> {
        Ok(())
    }
}
