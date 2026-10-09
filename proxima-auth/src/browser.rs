//! optional std-tier configuration and injected browser edge for oauth consent.

use bon::Builder;
use conflaguration::{Settings, Validate};
use serde::{Deserialize, Serialize};
use std::path::Path;

use crate::AuthorizationRequest;

/// controls whether an application asks its browser edge to open consent.
#[derive(Debug, Clone, PartialEq, Eq, Builder, Serialize, Deserialize, Settings)]
#[settings(prefix = "PROXIMA_AUTH_OAUTH")]
#[builder(derive(Clone, Debug))]
pub struct OAuthBrowserConfig {
    /// opens the provider consent url when true; false leaves it for manual presentation.
    #[setting(default = false)]
    #[serde(default)]
    #[builder(default)]
    pub launch: bool,
}

impl Default for OAuthBrowserConfig {
    fn default() -> Self {
        Self::builder().build()
    }
}

impl Validate for OAuthBrowserConfig {
    fn validate(&self) -> conflaguration::Result<()> {
        Ok(())
    }
}

impl OAuthBrowserConfig {
    /// loads layered browser policy from a configuration file and environment.
    ///
    /// # Errors
    /// returns an error when the file cannot be read or parsed, or the
    /// `PROXIMA_AUTH_OAUTH_LAUNCH` environment value cannot be parsed as a bool.
    #[must_use = "handle browser configuration loading failure"]
    pub fn from_path_then_env<ConfigPath: AsRef<Path>>(
        config_path: ConfigPath,
    ) -> conflaguration::Result<Self> {
        conflaguration::from_file_then_env(config_path.as_ref())
    }

    /// opens the consent url only when configured; the authorization request
    /// remains available for manual continuation when launch is disabled.
    #[must_use = "handle browser opener failure"]
    pub fn present<Opener: BrowserOpener>(
        &self,
        opener: &Opener,
        request: &AuthorizationRequest,
    ) -> Result<(), Opener::Error> {
        if self.launch {
            opener.open(request.url())?;
        }
        Ok(())
    }
}

/// caller-owned operating-system or embedded browser opening edge.
pub trait BrowserOpener {
    /// error returned when the host cannot open the authorization url.
    type Error;

    /// opens the url in the caller-selected browser environment.
    fn open(&self, url: &str) -> Result<(), Self::Error>;
}
