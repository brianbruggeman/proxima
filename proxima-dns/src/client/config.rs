//! `DnsResolverConfig` — the resolver client's config-mirror surface
//! (workspace principle 4): one type is the bon builder result, the serde
//! shape, and the conflaguration env surface (`DNS_RESOLVER_*`). Mirrors
//! `proxima_redis::client::config::RedisClientConfig`'s host/port split —
//! the live transport (a [`proxima_primitives::stream::DatagramFactory`])
//! is a runtime object injected at connect time, not in the config.

use std::net::{IpAddr, SocketAddr};

use bon::Builder;
use conflaguration::{Settings, Validate, ValidationMessage};
use serde::{Deserialize, Serialize};

fn default_resolver_ip() -> String {
    // 1.1.1.1 — a well-known public resolver (Cloudflare), used the same
    // way most stub-resolver libraries pick a working default: real,
    // reachable, and documented, not a placeholder. Deployments with their
    // own upstream override this field.
    "1.1.1.1".to_string()
}

fn default_port() -> u16 {
    53
}

fn default_query_timeout_ms() -> u64 {
    2_000
}

fn default_max_attempts() -> u32 {
    2
}

/// Resolver connection parameters for [`crate::client::DnsClientUpstream`].
/// Maps 1:1 to a TOML `[dns_resolver]` table or `DNS_RESOLVER_*` env vars,
/// and to the bon builder.
#[derive(Debug, Clone, PartialEq, Eq, Builder, Serialize, Deserialize, Settings)]
#[settings(prefix = "DNS_RESOLVER")]
#[builder(derive(Clone, Debug))]
pub struct DnsResolverConfig {
    /// Upstream resolver IP, dotted-decimal or IPv6 text form — an
    /// address, never a hostname: resolving a hostname to find your own
    /// resolver is circular. Kept as `String` (not `IpAddr`) so the
    /// conflaguration `DNS_RESOLVER_RESOLVER_IP` env var and TOML surface
    /// parse it the same way every other string setting does;
    /// [`Self::resolver_addr`] parses it at connect time.
    #[setting(default = "1.1.1.1")]
    #[serde(default = "default_resolver_ip")]
    #[builder(default = default_resolver_ip(), into)]
    pub resolver_ip: String,

    /// Resolver port (DNS default 53).
    #[setting(default = 53)]
    #[serde(default = "default_port")]
    #[builder(default = default_port())]
    pub port: u16,

    /// How long to wait for a matching reply before treating the query as
    /// timed out.
    #[setting(default = 2000)]
    #[serde(default = "default_query_timeout_ms")]
    #[builder(default = default_query_timeout_ms())]
    pub query_timeout_ms: u64,

    /// Send attempts before giving up (1 = no retry). UDP has no delivery
    /// guarantee, so a stub resolver client retries a lost query rather
    /// than surface a spurious timeout for one dropped packet.
    #[setting(default = 2)]
    #[serde(default = "default_max_attempts")]
    #[builder(default = default_max_attempts())]
    pub max_attempts: u32,
}

impl Default for DnsResolverConfig {
    fn default() -> Self {
        Self::builder().build()
    }
}

/// Errors parsing a `dns://` resolver DSN — mirrors the sibling client
/// crates' `KafkaConfigError`/`RedisConfigError` shape.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum DnsConfigError {
    #[error("dsn must start with dns://")]
    Scheme,
    #[error("dsn port must be a number")]
    Port,
}

impl DnsResolverConfig {
    /// Parses a `dns://resolver_ip[:port]` DSN — the resolver IP MUST be an
    /// IP literal, matching [`Self::resolver_ip`]'s own constraint (a stub
    /// resolver cannot resolve its own upstream's hostname). A missing
    /// field falls back to its default. This is the ergonomic entry the
    /// fluent `.dns(dsn)` client sugar lowers to.
    ///
    /// # Errors
    /// [`DnsConfigError::Scheme`] when the scheme is not `dns`,
    /// [`DnsConfigError::Port`] on a non-numeric port.
    pub fn from_dsn(dsn: &str) -> Result<Self, DnsConfigError> {
        let rest = dsn.strip_prefix("dns://").ok_or(DnsConfigError::Scheme)?;
        let (resolver_ip, port) = match rest.rsplit_once(':') {
            Some((host, port)) => (
                host,
                port.parse::<u16>().map_err(|_error| DnsConfigError::Port)?,
            ),
            None => (rest, default_port()),
        };
        let resolver_ip = if resolver_ip.is_empty() {
            default_resolver_ip()
        } else {
            resolver_ip.to_string()
        };
        Ok(Self {
            resolver_ip,
            port,
            ..Self::default()
        })
    }

    /// The socket address the client dials.
    ///
    /// # Errors
    /// [`crate::error::DnsClientError::Config`] if [`Self::resolver_ip`]
    /// isn't a parseable IP literal — checked here rather than at
    /// construction so a config loaded from an untrusted source fails at
    /// first use with a clear error instead of a builder-time panic. Reuses
    /// the crate's one client error type rather than minting a
    /// single-variant error type of its own (workspace principle 1).
    pub fn resolver_addr(&self) -> Result<SocketAddr, crate::error::DnsClientError> {
        self.resolver_ip
            .parse::<IpAddr>()
            .map(|ip| SocketAddr::new(ip, self.port))
            .map_err(|_| {
                crate::error::DnsClientError::Config(format!(
                    "resolver_ip {:?} is not a valid IP address literal",
                    self.resolver_ip
                ))
            })
    }
}

impl Validate for DnsResolverConfig {
    fn validate(&self) -> conflaguration::Result<()> {
        let mut errors = Vec::new();
        if self.resolver_ip.parse::<IpAddr>().is_err() {
            errors.push(ValidationMessage::new(
                "resolver_ip",
                "must be a valid IP address literal, not a hostname",
            ));
        }
        if self.port == 0 {
            errors.push(ValidationMessage::new("port", "must be non-zero"));
        }
        if self.query_timeout_ms == 0 {
            errors.push(ValidationMessage::new(
                "query_timeout_ms",
                "must be non-zero",
            ));
        }
        if self.max_attempts == 0 {
            errors.push(ValidationMessage::new(
                "max_attempts",
                "must be at least 1",
            ));
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(conflaguration::Error::Validation { errors })
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn default_matches_builder() {
        assert_eq!(DnsResolverConfig::default(), DnsResolverConfig::builder().build());
        let config = DnsResolverConfig::default();
        assert_eq!(config.port, 53);
        assert_eq!(config.resolver_addr().unwrap().to_string(), "1.1.1.1:53");
    }

    #[test]
    fn builder_overrides_defaults() {
        let config = DnsResolverConfig::builder()
            .resolver_ip("9.9.9.9")
            .port(5353)
            .max_attempts(3)
            .build();
        assert_eq!(config.resolver_addr().unwrap().to_string(), "9.9.9.9:5353");
        assert_eq!(config.max_attempts, 3);
    }

    #[test]
    fn dsn_full_round_trips_resolver_ip_and_port() {
        let config = DnsResolverConfig::from_dsn("dns://9.9.9.9:5353").unwrap();
        assert_eq!(config.resolver_ip, "9.9.9.9");
        assert_eq!(config.port, 5353);
    }

    #[test]
    fn dsn_minimal_falls_back_to_the_default_port() {
        let config = DnsResolverConfig::from_dsn("dns://9.9.9.9").unwrap();
        assert_eq!(config.port, 53);
    }

    #[test]
    fn dsn_rejects_foreign_scheme() {
        assert_eq!(
            DnsResolverConfig::from_dsn("redis://9.9.9.9"),
            Err(DnsConfigError::Scheme)
        );
    }

    #[test]
    fn dsn_rejects_non_numeric_port() {
        assert_eq!(
            DnsResolverConfig::from_dsn("dns://9.9.9.9:notaport"),
            Err(DnsConfigError::Port)
        );
    }

    #[test]
    fn resolver_addr_rejects_a_hostname() {
        let config = DnsResolverConfig::builder().resolver_ip("resolver.example.com").build();
        let error = config.resolver_addr().unwrap_err();
        assert!(error.to_string().contains("resolver.example.com"));
    }

    #[test]
    fn validate_rejects_a_hostname() {
        let config = DnsResolverConfig::builder().resolver_ip("resolver.example.com").build();
        assert!(config.validate().is_err());
    }

    #[test]
    fn validate_rejects_zero_port() {
        let config = DnsResolverConfig::builder().port(0).build();
        assert!(config.validate().is_err());
    }

    #[test]
    fn validate_rejects_zero_attempts() {
        let config = DnsResolverConfig::builder().max_attempts(0).build();
        assert!(config.validate().is_err());
    }

    #[test]
    fn config_round_trips_through_serde() {
        let config = DnsResolverConfig::builder().port(5353).build();
        let json = serde_json::to_string(&config).expect("ser");
        let back: DnsResolverConfig = serde_json::from_str(&json).expect("de");
        assert_eq!(config, back);
    }
}
