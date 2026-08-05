//! `DnsServerConfig` — the listener facade's config-mirror surface
//! (workspace principle 4): one type is the bon builder result, the serde
//! shape, and the conflaguration env surface (`DNS_*`). Mirrors
//! `proxima_redis::config::RedisServerConfig`.

use bon::Builder;
use conflaguration::{Settings, Validate, ValidationMessage};
use proxima_core::ProximaError;
use serde::{Deserialize, Serialize};
use serde_json::Value;

fn default_max_message() -> usize {
    65_535
}

/// Resolve the per-connection config a listener spec's `dns` object asks for,
/// falling back to `base` when the spec says nothing. Both `.any()`
/// candidates resolve identically — a spec that overrides the message cap for
/// DNS-over-TCP means the same thing for DNS-over-UDP on the same bind.
///
/// # Errors
/// [`ProximaError::Config`] when the `dns` object is present but does not
/// deserialize into a [`DnsServerConfig`].
pub(crate) fn resolve_config(
    base: &DnsServerConfig,
    spec: &Value,
) -> Result<DnsServerConfig, ProximaError> {
    match spec.get("dns") {
        None => Ok(base.clone()),
        Some(overrides) => serde_json::from_value(overrides.clone())
            .map_err(|error| ProximaError::Config(format!("dns spec: {error}"))),
    }
}

/// DNS wire-server configuration, shared by [`crate::DnsDatagramProtocol`]
/// (UDP) and [`crate::DnsAnyProtocol`] (TCP).
#[derive(Debug, Clone, PartialEq, Eq, Builder, Serialize, Deserialize, Settings)]
#[settings(prefix = "DNS")]
#[builder(derive(Clone, Debug))]
pub struct DnsServerConfig {
    /// Hard cap on one DNS message (query or response), the DoS guard both
    /// listeners enforce before parsing. RFC 1035 §4.2.2 lets a TCP message
    /// declare up to 65535 bytes (the 2-byte length prefix's max value);
    /// UDP is already message-atomic (one datagram, one message) and never
    /// exceeds the OS datagram size, so this cap mainly bites the TCP path.
    #[setting(default = 65535)]
    #[serde(default = "default_max_message")]
    #[builder(default = default_max_message())]
    pub max_message_bytes: usize,
}

impl Default for DnsServerConfig {
    fn default() -> Self {
        Self::builder().build()
    }
}

impl Validate for DnsServerConfig {
    fn validate(&self) -> conflaguration::Result<()> {
        let mut errors = Vec::new();
        // 12 bytes is the fixed header alone (RFC 1035 §4.1.1) — anything
        // smaller can never hold a legal message.
        if self.max_message_bytes < 12 {
            errors.push(ValidationMessage::new(
                "max_message_bytes",
                "must be at least 12 bytes (the fixed DNS header)",
            ));
        }
        if self.max_message_bytes > 65535 {
            errors.push(ValidationMessage::new(
                "max_message_bytes",
                "must be at most 65535 bytes (RFC 1035 §4.2.2 TCP length-prefix range)",
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
    fn default_config_is_valid() {
        let config = DnsServerConfig::default();
        assert!(config.validate().is_ok());
        assert_eq!(config.max_message_bytes, 65_535);
    }

    #[test]
    fn default_matches_builder() {
        assert_eq!(
            DnsServerConfig::default(),
            DnsServerConfig::builder().build()
        );
    }

    #[test]
    fn builder_overrides_defaults() {
        let config = DnsServerConfig::builder().max_message_bytes(512).build();
        assert_eq!(config.max_message_bytes, 512);
    }

    #[test]
    fn validate_rejects_below_header_size() {
        let config = DnsServerConfig::builder().max_message_bytes(11).build();
        assert!(config.validate().is_err());
    }

    #[test]
    fn validate_rejects_above_u16_max() {
        let config = DnsServerConfig::builder().max_message_bytes(70_000).build();
        assert!(config.validate().is_err());
    }

    #[test]
    fn config_round_trips_through_serde() {
        let config = DnsServerConfig::builder().max_message_bytes(4096).build();
        let json = serde_json::to_string(&config).expect("ser");
        let back: DnsServerConfig = serde_json::from_str(&json).expect("de");
        assert_eq!(config, back);
    }

    #[test]
    fn resolve_config_overrides_max_message_bytes_from_the_spec() {
        let base = DnsServerConfig::default();
        let spec = serde_json::json!({ "dns": { "max_message_bytes": 4096 } });
        let resolved = resolve_config(&base, &spec).expect("spec resolves");
        assert_eq!(resolved.max_message_bytes, 4096);
    }

    #[test]
    fn resolve_config_falls_back_to_the_base_config_with_no_spec_override() {
        let base = DnsServerConfig::default();
        let resolved = resolve_config(&base, &Value::Null).expect("no override resolves");
        assert_eq!(resolved, base);
    }

    #[test]
    fn resolve_config_rejects_a_malformed_dns_object() {
        let base = DnsServerConfig::default();
        let spec = serde_json::json!({ "dns": { "max_message_bytes": "not a number" } });
        assert!(resolve_config(&base, &spec).is_err());
    }
}
